import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
processed_dir = PROJECT_ROOT / "data" / "processed"

X = pd.read_csv(processed_dir / "Parkinsons_v2_features.csv")
y = pd.read_csv(processed_dir / "Parkinsons_v2_status.csv").iloc[:, 0]
groups = pd.read_csv(processed_dir / "Parkinsons_v2_groups.csv").iloc[:, 0]

cv = GroupKFold(n_splits=5)
scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(recall_score, pos_label=0),
}

print("1. Testing PCA (30 components) + SMOTE + XGBoost...")
pipeline_pca_xgb = make_imblearn_pipeline(
    StandardScaler(),
    PCA(n_components=30, random_state=42),
    SMOTE(random_state=42),
    XGBClassifier(eval_metric='logloss', random_state=42)
)
results_pca_xgb = cross_validate(pipeline_pca_xgb, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
print(f"BalAcc: {np.mean(results_pca_xgb['test_balanced_accuracy']):.3f} | Sens: {np.mean(results_pca_xgb['test_sensitivity']):.3f} | Spec: {np.mean(results_pca_xgb['test_specificity']):.3f}\n")

print("2. Testing PCA (0.95 variance) + SPW + XGBoost (No SMOTE)...")
pipeline_pca_spw = make_imblearn_pipeline(
    StandardScaler(),
    PCA(n_components=0.95, random_state=42),
    XGBClassifier(scale_pos_weight=0.34, eval_metric='logloss', random_state=42)
)
results_pca_spw = cross_validate(pipeline_pca_spw, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
print(f"BalAcc: {np.mean(results_pca_spw['test_balanced_accuracy']):.3f} | Sens: {np.mean(results_pca_spw['test_sensitivity']):.3f} | Spec: {np.mean(results_pca_spw['test_specificity']):.3f}\n")

print("3. Testing Soft Voting Classifier (XGB + RF + LR)...")
clf1 = XGBClassifier(eval_metric='logloss', random_state=42)
clf2 = RandomForestClassifier(class_weight='balanced', random_state=42)
clf3 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

voting_clf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)], voting='soft')

pipeline_voting = make_imblearn_pipeline(
    StandardScaler(),
    PCA(n_components=30, random_state=42),
    SMOTE(random_state=42),
    voting_clf
)
results_voting = cross_validate(pipeline_voting, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
print(f"BalAcc: {np.mean(results_voting['test_balanced_accuracy']):.3f} | Sens: {np.mean(results_voting['test_sensitivity']):.3f} | Spec: {np.mean(results_voting['test_specificity']):.3f}\n")
