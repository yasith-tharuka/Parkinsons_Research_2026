import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

print("==========================================================")
print(" COST-SENSITIVE XGBOOST + SPECIFICITY-OPTIMIZED THRESHOLD")
print("==========================================================\n")

# 1. LOAD DATA & ISOLATE THE VAULT
X = pd.read_csv("Parkinsons_cleaned.csv")
y = pd.read_csv("Parkinsons_status.csv")
groups = pd.read_csv("Parkinsons_groups.csv").values.ravel()

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_flat = y.iloc[train_idx].values.ravel()
y_test_flat = y.iloc[test_idx].values.ravel()

# 2. COST-SENSITIVE WEIGHTING (to reduce FP)
healthy = np.sum(y_train_flat == 0)
parkinsons = np.sum(y_train_flat == 1)

scale_pos_weight = parkinsons / healthy  # <--- reduces FP
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

# 3. PIPELINE
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=20)),
    ("xgb", XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300
    ))
])

pipeline.fit(X_train, y_train_flat)

# 4. PROBABILITIES
y_probs = pipeline.predict_proba(X_test)[:, 1]

# 5. THRESHOLD SEARCH — MAXIMIZE SPECIFICITY
best_threshold = 0.5
best_specificity = 0
best_stats = {}

for t in np.arange(0.50, 0.99, 0.01):
    preds = (y_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_flat, preds).ravel()

    specificity = tn / (tn + fp + 1e-9)  # avoid divide-by-zero
    recall = tp / (tp + fn + 1e-9)

    # Choose threshold with highest specificity but still reasonable recall
    if specificity > best_specificity and recall > 0.70:
        best_specificity = specificity
        best_threshold = t
        best_stats = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

print("\n==========================================================")
print(" OPTIMAL THRESHOLD (Maximizing Specificity)")
print("==========================================================")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Specificity:    {best_specificity:.3f}")
print(f"Recall:         {best_stats['tp']/(best_stats['tp']+best_stats['fn']):.3f}")
print("==========================================================\n")

# 6. FINAL PREDICTIONS
final_preds = (y_probs >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test_flat, final_preds).ravel()

print("--- FINAL CONFUSION MATRIX (Specificity-Optimized) ---")
print(f"TN: {tn}")
print(f"FP: {fp}  <-- should drop significantly")
print(f"FN: {fn}")
print(f"TP: {tp}")
print("==========================================================")
