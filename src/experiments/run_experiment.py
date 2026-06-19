import pandas as pd
import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Suppress minor matplotlib warnings for a clean console
warnings.filterwarnings("ignore")

# CRITICAL IMPORTS FOR THE GOLD-STANDARD STRATEGY
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
)

def choose_operating_threshold(
    y_true,
    y_prob,
    target_sensitivity=1.00,
    min_specificity=0.25,
    min_sensitivity_floor=0.90,
    sensitivity_step=0.01,
):
    """Pick a threshold with medical priority: sensitivity first, but keep a minimum specificity."""
    candidate_thresholds = np.linspace(0.0, 1.0, 1001)
    all_candidates = []

    for threshold in candidate_thresholds:
        preds = (y_prob >= threshold).astype(int)
        sensitivity = recall_score(y_true, preds, pos_label=1, zero_division=0)
        specificity = recall_score(y_true, preds, pos_label=0, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, preds)

        all_candidates.append(
            {
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'balanced_accuracy': balanced_acc,
            }
        )

    def pick_best(candidates, policy, target_used):
        return max(
            candidates,
            key=lambda item: (
                item['specificity'],
                item['balanced_accuracy'],
                item['threshold'],
            ),
        )

    # First, try strict target sensitivity with a specificity safety floor.
    strict_candidates = [
        row
        for row in all_candidates
        if row['sensitivity'] >= target_sensitivity and row['specificity'] >= min_specificity
    ]
    if strict_candidates:
        best = pick_best(strict_candidates, 'strict_target_with_specificity_floor', target_sensitivity)
        best['policy'] = 'strict_target_with_specificity_floor'
        best['target_sensitivity_used'] = target_sensitivity
        return best

    # If strict target is impossible, relax sensitivity gradually but keep specificity floor.
    current_target = target_sensitivity
    while current_target >= min_sensitivity_floor:
        adaptive_candidates = [
            row
            for row in all_candidates
            if row['sensitivity'] >= current_target and row['specificity'] >= min_specificity
        ]
        if adaptive_candidates:
            best = pick_best(adaptive_candidates, 'adaptive_target_with_specificity_floor', current_target)
            best['policy'] = 'adaptive_target_with_specificity_floor'
            best['target_sensitivity_used'] = current_target
            return best
        current_target = round(current_target - sensitivity_step, 10)

    # Final fallback: maximize sensitivity, then specificity.
    best = max(
        all_candidates,
        key=lambda item: (
            item['sensitivity'],
            item['specificity'],
            item['balanced_accuracy'],
            item['threshold'],
        ),
    )
    best['policy'] = 'best_possible_sensitivity'
    best['target_sensitivity_used'] = best['sensitivity']
    return best

def run_experiment():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    MIN_PARKINSON_PROBABILITY = 0.25

    # PART 1: (Data Loading & Group-Aware Split) ----------------------------------------------------
    X = pd.read_csv(DATA_DIR / "Parkinsons_cleaned.csv")
    y = pd.read_csv(DATA_DIR / "Parkinsons_status.csv")
    groups = pd.read_csv(DATA_DIR / "Parkinsons_groups.csv").values.ravel() 

    # Isolate a final test "Vault" (20% of patients)
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx] 

    y_train_flat = y_train.values.ravel()
    y_test_flat = y_test.values.ravel()

    print(f"Training Set (Instances): {X_train.shape[0]}")
    print(f"Testing Set (The Vault): {X_test.shape[0]}")
    print("--------------------------------------------------")

    # PART 2: ROBUST STRATEGY SELECTION + GROUP-CV ----------------------------------------------
    print("Initializing candidate pipelines...")

    healthy_count = int(np.sum(y_train_flat == 0))
    parkinsons_count = int(np.sum(y_train_flat == 1))
    scale_pos_weight_balanced = healthy_count / parkinsons_count if parkinsons_count else 1.0
    medical_scale_pos_weight = max(1.0, scale_pos_weight_balanced)

    print(
        f"Training Class Distribution -> Healthy (0): {healthy_count}, "
        f"Parkinson's (1): {parkinsons_count}"
    )
    print(f"Balanced scale_pos_weight: {scale_pos_weight_balanced:.3f}")
    print(f"Medical-priority scale_pos_weight: {medical_scale_pos_weight:.3f}")

    xgb_common_params = {
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_estimators': 250,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'n_jobs': 1,
    }

    candidate_pipelines = {
        'SMOTE + XGBoost': make_imblearn_pipeline(
            StandardScaler(),
            SMOTE(random_state=42),
            XGBClassifier(**xgb_common_params),
        ),
        'Weighted XGBoost (No SMOTE)': make_imblearn_pipeline(
            StandardScaler(),
            XGBClassifier(**xgb_common_params, scale_pos_weight=medical_scale_pos_weight),
        ),
    }

    # THE LEAKAGE-FREE FIX: 5-Fold Group Cross Validation
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        print("Using StratifiedGroupKFold for better per-fold class balance.")
    except ImportError:
        cv_strategy = GroupKFold(n_splits=5)
        print("StratifiedGroupKFold unavailable, falling back to GroupKFold.")

    specificity_scorer = make_scorer(recall_score, pos_label=0)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'f1_macro': 'f1_macro',
        'roc_auc': 'roc_auc',
        'sensitivity': 'recall',
        'specificity': specificity_scorer,
        'balanced_accuracy': 'balanced_accuracy',
    }

    print("Running 5-Fold Group Cross-Validation for each strategy... (Please wait)\n")
    strategy_results = {}

    for strategy_name, pipeline in candidate_pipelines.items():
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train_flat,
            groups=groups_train,
            cv=cv_strategy,
            scoring=scoring_metrics,
            n_jobs=-1,
        )

        strategy_results[strategy_name] = {
            'pipeline': pipeline,
            'cv': cv_result,
            'sensitivity': statistics.mean(cv_result['test_sensitivity']),
            'balanced_accuracy': statistics.mean(cv_result['test_balanced_accuracy']),
            'specificity': statistics.mean(cv_result['test_specificity']),
            'f1_macro': statistics.mean(cv_result['test_f1_macro']),
        }

        print(f"[{strategy_name}] CV Accuracy: {statistics.mean(cv_result['test_accuracy']):.3f}")
        print(f"[{strategy_name}] CV F1-Score: {statistics.mean(cv_result['test_f1']):.3f}")
        print(f"[{strategy_name}] CV ROC-AUC:  {statistics.mean(cv_result['test_roc_auc']):.3f}")
        print(f"[{strategy_name}] CV Macro F1: {statistics.mean(cv_result['test_f1_macro']):.3f}")
        print(
            f"[{strategy_name}] CV Balanced Accuracy: "
            f"{statistics.mean(cv_result['test_balanced_accuracy']):.3f}"
        )
        print(
            f"[{strategy_name}] CV Sensitivity (Parkinson's Recall): "
            f"{statistics.mean(cv_result['test_sensitivity']):.3f}"
        )
        print(
            f"[{strategy_name}] CV Specificity (Healthy Recall): "
            f"{statistics.mean(cv_result['test_specificity']):.3f}\n"
        )

    selected_strategy, selected_result = max(
        strategy_results.items(),
        key=lambda item: (
            item[1]['sensitivity'],
            item[1]['specificity'],
            item[1]['balanced_accuracy'],
            item[1]['f1_macro'],
        ),
    )

    champion_pipeline = selected_result['pipeline']
    cv_results = selected_result['cv']

    print(f"Selected Strategy: {selected_strategy}")
    print(f"Champion CV Accuracy: {statistics.mean(cv_results['test_accuracy']):.3f}")
    print(f"Champion CV F1-Score: {statistics.mean(cv_results['test_f1']):.3f}")
    print(f"Champion CV ROC-AUC:  {statistics.mean(cv_results['test_roc_auc']):.3f}")
    print(f"Champion CV Macro F1: {statistics.mean(cv_results['test_f1_macro']):.3f}")
    print(f"Champion CV Balanced Accuracy: {statistics.mean(cv_results['test_balanced_accuracy']):.3f}")
    print(f"Champion CV Sensitivity (Parkinson's Recall): {statistics.mean(cv_results['test_sensitivity']):.3f}")
    print(f"Champion CV Specificity (Healthy Recall): {statistics.mean(cv_results['test_specificity']):.3f}")

    print("\nCalibrating diagnosis threshold from out-of-fold predictions...")
    oof_probs = cross_val_predict(
        champion_pipeline,
        X_train,
        y_train_flat,
        groups=groups_train,
        cv=cv_strategy,
        method='predict_proba',
        n_jobs=-1,
    )[:, 1]
    threshold_result = choose_operating_threshold(
        y_train_flat,
        oof_probs,
        target_sensitivity=1.00,
        min_specificity=0.25,
        min_sensitivity_floor=0.90,
        sensitivity_step=0.01,
    )
    custom_threshold = threshold_result['threshold']
    raw_threshold = custom_threshold

    # Avoid overly permissive thresholds that classify nearly everyone as Parkinson's.
    if custom_threshold < MIN_PARKINSON_PROBABILITY:
        custom_threshold = MIN_PARKINSON_PROBABILITY

    print(f"Selected data-driven threshold: {raw_threshold:.3f}")
    if custom_threshold != raw_threshold:
        print(
            f"Applied clinical threshold floor: {MIN_PARKINSON_PROBABILITY:.3f} "
            f"(final threshold: {custom_threshold:.3f})"
        )
    print(
        f"Threshold calibration policy: {threshold_result['policy']} | "
        f"Target Sensitivity Used: {threshold_result['target_sensitivity_used']:.3f} | "
        f"CV Sensitivity: {threshold_result['sensitivity']:.3f} | "
        f"CV Specificity: {threshold_result['specificity']:.3f}"
    )
    print("--------------------------------------------------")

    # PART 3: FINAL TEST SET EVALUATION -------------------------------------------------------------
    print("\n--- Final Real-World Evaluation ---")
    champion_pipeline.fit(X_train, y_train_flat)

    # Extract the exact probability (0% to 100%) that the patient has Parkinson's
    probs = champion_pipeline.predict_proba(X_test)[:, 1]

    # Apply calibrated threshold learned only from training folds.
    final_preds = (probs >= custom_threshold).astype(int)

    final_accuracy = accuracy_score(y_test_flat, final_preds)
    final_f1 = f1_score(y_test_flat, final_preds, zero_division=0)
    final_balanced_acc = balanced_accuracy_score(y_test_flat, final_preds)
    final_auc = roc_auc_score(y_test_flat, probs)
    final_specificity = recall_score(y_test_flat, final_preds, pos_label=0, zero_division=0)
    final_sensitivity = recall_score(y_test_flat, final_preds, pos_label=1, zero_division=0)

    print(f"Applied Threshold: {custom_threshold:.3f}")
    print(f"Final Accuracy: {final_accuracy:.3f}")
    print(f"Final F1-Score: {final_f1:.3f}")
    print(f"Final Balanced Accuracy: {final_balanced_acc:.3f}")
    print(f"Final ROC-AUC: {final_auc:.3f}")
    print(f"Final Sensitivity (Parkinson's Recall): {final_sensitivity:.3f}")
    print(f"Final Specificity (Healthy Recall): {final_specificity:.3f}\n")
    print(
        classification_report(
            y_test_flat,
            final_preds,
            target_names=["Healthy (0)", "Parkinson's (1)"],
            zero_division=0,
        )
    )

    # FIGURE 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test_flat, final_preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', "Parkinson's"], yticklabels=['Healthy', "Parkinson's"])
    plt.title(f"Final Real-World Performance: {selected_strategy} (GroupCV + Calibrated Threshold)")
    plt.ylabel('True Diagnosis')
    plt.xlabel('Classifier Prediction')
    plt.tight_layout()
    matrix_output_path = FIGURES_DIR / "FINAL_Winning_Matrix.png"
    plt.savefig(matrix_output_path, dpi=300)
    print(f"Figure 1 generated: '{matrix_output_path}'")

    # PART 4: NOVELTY (FEATURE IMPORTANCE) ----------------------------------------------------------
    print("\nExtracting Explainable AI Biomarkers...")
    xgb_model = champion_pipeline.named_steps['xgbclassifier']
    importances = xgb_model.feature_importances_

    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # FIGURE 2: XAI Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis', hue='Feature', legend=False)
    plt.title(f"Top 10 Most Critical Vocal Biomarkers ({selected_strategy})")
    plt.xlabel("Relative Importance to the Model")
    plt.ylabel("Acoustic Feature")
    plt.tight_layout()
    importance_output_path = FIGURES_DIR / "Feature_Importance.png"
    plt.savefig(importance_output_path, dpi=300)
    print(f"Figure 2 generated: '{importance_output_path}'")
    print("\nTop 5 Drivers of Diagnosis:")
    print(importance_df.head(5).to_string(index=False))
    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()
