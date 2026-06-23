import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score

def run_v2_experiment():
    print("==================================================")
    print("   RUNNING V2 EXPERIMENT ON HIGH-QUALITY DATASET  ")
    print("==================================================")
    print("Dataset: UCI Parkinson's Disease Classification")
    print("Features: 754 acoustic features")
    print("--------------------------------------------------")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    # Load the new preprocessed data
    X = pd.read_csv(processed_dir / "Parkinsons_v2_features.csv")
    y = pd.read_csv(processed_dir / "Parkinsons_v2_status.csv").iloc[:, 0]
    groups = pd.read_csv(processed_dir / "Parkinsons_v2_groups.csv").iloc[:, 0]
    
    print(f"Total instances: {X.shape[0]}")
    print(f"Unique patients (groups): {groups.nunique()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Define our pipeline: Scale -> SMOTE -> XGBoost
    pipeline = make_imblearn_pipeline(
        StandardScaler(),
        SMOTE(random_state=42),
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    )
    
    # Patient-aware cross-validation to prevent leakage
    cv = GroupKFold(n_splits=5)
    
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(recall_score, pos_label=0),
    }
    
    print("\nRunning Patient-Aware Cross Validation (GroupKFold)...")
    results = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("\n--- RESULTS ACROSS 5 FOLDS ---")
    print(f"Average Balanced Accuracy : {np.mean(results['test_balanced_accuracy']):.3f} +/- {np.std(results['test_balanced_accuracy']):.3f}")
    print(f"Average Sensitivity       : {np.mean(results['test_sensitivity']):.3f} +/- {np.std(results['test_sensitivity']):.3f}")
    print(f"Average Specificity       : {np.mean(results['test_specificity']):.3f} +/- {np.std(results['test_specificity']):.3f}")
    print("==================================================")

if __name__ == "__main__":
    run_v2_experiment()
